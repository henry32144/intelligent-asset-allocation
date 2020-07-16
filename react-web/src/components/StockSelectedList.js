import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import List from '@material-ui/core/List';
import Box from '@material-ui/core/Box';
import ListSubheader from '@material-ui/core/ListSubheader';
import StockListItem from './StockListItem'
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import { FixedSizeList } from 'react-window';

const useStyles = makeStyles((theme) => ({
  root: {},
  stockComponent: {},
  listSubHeader: {
    textAlign: 'initial'
  }
}));


function renderRow(props) {
  const { data, index, style } = props;
  const rowItem = data[index];
  return (
    <div style={style}>
      {
        <StockListItem companyName={rowItem.companyName} companyCode={rowItem.companyCode}>
        </StockListItem>
      }
    </div>
  );
}


function StockSelectedList(props) {
  const { selectedStocks, additionalStyles } = props
  const classes = useStyles();

  return (
    <Box className={additionalStyles.stockComponent}>
      <FixedSizeList
        height={350}
        itemSize={60}
        itemCount={selectedStocks.length}
        itemData={selectedStocks}
      >
        {renderRow}
      </FixedSizeList >
    </Box>
  );
}

export default StockSelectedList