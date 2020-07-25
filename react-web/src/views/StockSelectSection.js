import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import Box from '@material-ui/core/Box';
import StockSelectedList from '../components/StockSelectedList'

const useStyles = makeStyles((theme) => ({
  root: {
    height: 'calc(100% - 16px)',
    padding: theme.spacing(1, 1, 1),
    flexGrow: 1,
  },
}));

export default function StockSelectSection(props) {
  const classes = useStyles();

  return (
    <Box className={classes.root} boxShadow={1}>
      <StockSelectedList
        selectedStocks={props.selectedStocks}
        setSelectedStocks={props.setSelectedStocks} />
    </Box>
  );
}