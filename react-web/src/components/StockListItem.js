import React from 'react';
import { makeStyles } from '@material-ui/core/styles';
import ListItem from '@material-ui/core/ListItem';
import ListItemText from '@material-ui/core/ListItemText';
import IconButton from '@material-ui/core/IconButton';
import DeleteIcon from '@material-ui/icons/Delete';

const useStyles = makeStyles((theme) => ({
}));

export default function StockListItem(props) {
  /* 
    props data structure-
      companyName(string): the name of the company, Ex: Apple Inc.
      companySymbol(string): the symbol of the company in the market, Ex: NASDAQ: AAPL
  */
  const classes = useStyles();

  const deleteButtonOnClick = () => {
    props.removeSelectedStock(props.companyId);
  };


  return (
    <div className={classes.root}>
      <ListItem alignItems="flex-start">
        <ListItemText primary={props.companyName} secondary={props.companySymbol} />
        <IconButton edge="end" aria-label="delete" onClick={deleteButtonOnClick}>
          <DeleteIcon />
        </IconButton>
      </ListItem>
    </div>
  );
}